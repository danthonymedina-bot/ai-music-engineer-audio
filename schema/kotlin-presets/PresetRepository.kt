package com.david.aimusicengineer.presets

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.Locale
import java.util.UUID

class PresetNameConflictException(val name: String, val existingId: String) :
    IllegalStateException("a preset named \"$name\" already exists")
class PresetNotFoundException(key: String) : NoSuchElementException("no preset $key")
class InvalidPresetException(msg: String, cause: Throwable? = null) : IllegalArgumentException(msg, cause)

/** Local-first store of Freestyle presets. All functions are main-safe suspend functions. */
interface PresetRepository {
    /** Observes the sorted list (case-insensitive by name). Emits after every change. */
    val presets: StateFlow<List<FreestylePreset>>
    /** File names moved to quarantine because they could not be read (for an optional UI notice). */
    val quarantined: StateFlow<List<String>>

    suspend fun save(name: String, params: PresetParams, stems: Map<String, JsonObject>? = null,
                     overwrite: Boolean = false): FreestylePreset
    suspend fun list(): List<FreestylePreset>
    suspend fun load(id: String): FreestylePreset?
    suspend fun loadByName(name: String): FreestylePreset?
    suspend fun rename(id: String, newName: String): FreestylePreset
    suspend fun delete(id: String): Boolean
    suspend fun exportJson(id: String): String
    /** Imports a shared preset. Name conflicts: overwrite=true replaces it, otherwise " (2)", " (3)"... is added. */
    suspend fun importJson(json: String, overwrite: Boolean = false): FreestylePreset
}

/** Version migration hook: fromVersion -> transform producing fromVersion+1. */
fun interface PresetMigration { fun migrate(old: JsonObject): JsonObject }

object PresetMigrations {
    /**
     * v0 = early stand-in format (no schemaVersion): {"name", "eqGains":[5 dB values], "drive", "widthPct"}.
     * Add an entry here whenever FreestylePreset.CURRENT_SCHEMA_VERSION is bumped.
     */
    val default: Map<Int, PresetMigration> = mapOf(0 to PresetMigration { o ->
        val gains = (o["eqGains"] as? kotlinx.serialization.json.JsonArray)?.map { it.jsonPrimitive.content.toDouble() }
            ?: emptyList()
        fun g(i: Int) = gains.getOrElse(i) { 0.0 }
        fun num(k: String) = (o[k] as? JsonPrimitive)?.content?.toDoubleOrNull()
        val drive = num("drive") ?: 0.0
        val width = num("widthPct") ?: 100.0
        val p = PresetParams(
            eq = Eq(Eq.fixedBands(g(0), g(1), g(2), g(3), g(4))),
            saturation = Saturation(enabled = drive > 0, driveDb = drive),
            width = Width(enabled = width != 100.0, widthPct = width))
        val m = o.toMutableMap()
        listOf("eqGains", "drive", "widthPct").forEach { m.remove(it) }
        m["params"] = PresetJson.encodeToJsonElement(p)
        m["schemaVersion"] = JsonPrimitive(1)
        JsonObject(m)
    })
}

/**
 * One JSON file per preset in [dir] (use context.filesDir/"presets"). Writes are atomic (temp file + atomic move),
 * so a crash never leaves a half-written preset; unreadable files are moved to [dir]/corrupt and reported.
 * Jetpack DataStore is not required; if you prefer it, keep this class's semantics and swap the storage.
 */
class FilePresetRepository(
    private val dir: File,
    private val clock: () -> Long = System::currentTimeMillis,
    private val newId: () -> String = { UUID.randomUUID().toString() },
    private val migrations: Map<Int, PresetMigration> = PresetMigrations.default,
) : PresetRepository {
    private val mutex = Mutex()
    private val cache = LinkedHashMap<String, FreestylePreset>()
    private val _presets = MutableStateFlow<List<FreestylePreset>>(emptyList())
    private val _quarantined = MutableStateFlow<List<String>>(emptyList())
    override val presets: StateFlow<List<FreestylePreset>> = _presets.asStateFlow()
    override val quarantined: StateFlow<List<String>> = _quarantined.asStateFlow()
    private var loaded = false

    private fun key(name: String) = name.trim().lowercase(Locale.ROOT)
    private fun cleanName(name: String): String {
        val n = name.trim().replace(Regex("\\s+"), " ")
        if (n.isEmpty() || n.length > 64) throw InvalidPresetException("name must be 1..64 characters")
        return n
    }

    private fun sorted() = cache.values.sortedWith(compareBy({ key(it.name) }, { it.createdAt }, { it.id }))
    private fun publish() { _presets.value = sorted() }
    private fun byName(name: String) = cache.values.firstOrNull { key(it.name) == key(name) }

    /** Parses (and migrates/sanitizes) a JSON document into a preset. */
    internal fun decode(text: String): Pair<FreestylePreset, Boolean> {
        var o: JsonObject = try { PresetJson.parseToJsonElement(text).jsonObject }
            catch (e: Exception) { throw InvalidPresetException("not a JSON object", e) }
        var v = (o["schemaVersion"] as? JsonPrimitive)?.intOrNull ?: 0
        val migrated = v != FreestylePreset.CURRENT_SCHEMA_VERSION
        if (v > FreestylePreset.CURRENT_SCHEMA_VERSION)
            throw InvalidPresetException("preset schemaVersion $v is newer than this app (${FreestylePreset.CURRENT_SCHEMA_VERSION})")
        while (v < FreestylePreset.CURRENT_SCHEMA_VERSION) {
            val m = migrations[v] ?: throw InvalidPresetException("no migration from schemaVersion $v")
            o = m.migrate(o)
            v = (o["schemaVersion"] as? JsonPrimitive)?.int ?: (v + 1)
        }
        val fill = o.toMutableMap()
        if (fill["id"] == null) fill["id"] = JsonPrimitive(newId())
        val now = clock()
        if (fill["createdAt"] == null) fill["createdAt"] = JsonPrimitive(now)
        if (fill["updatedAt"] == null) fill["updatedAt"] = fill["createdAt"]!!
        val p = try { PresetJson.decodeFromJsonElement<FreestylePreset>(JsonObject(fill)) }
            catch (e: Exception) { throw InvalidPresetException("invalid preset: ${e.message}", e) }
        if (runCatching { UUID.fromString(p.id) }.isFailure) throw InvalidPresetException("id must be a UUID")
        val s = PresetValidator.sanitize(p.params)
        return p.copy(name = cleanName(p.name), params = s.params) to (migrated || s.warnings.isNotEmpty())
    }

    private fun fileOf(id: String) = File(dir, "$id.json")

    private fun writeAtomic(p: FreestylePreset) {
        dir.mkdirs()
        val tmp = File(dir, ".${p.id}.json.tmp")
        tmp.outputStream().use { out ->
            out.write(PresetJson.encodeToString(FreestylePreset.serializer(), p).toByteArray(Charsets.UTF_8))
            out.fd.sync()
        }
        try {
            Files.move(tmp.toPath(), fileOf(p.id).toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } catch (e: AtomicMoveNotSupportedException) {
            Files.move(tmp.toPath(), fileOf(p.id).toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun quarantine(f: File) {
        val q = File(dir, "corrupt").apply { mkdirs() }
        val dest = File(q, "${f.name}.${clock()}")
        f.renameTo(dest)
        _quarantined.value = _quarantined.value + f.name
    }

    private fun ensureLoaded() {
        if (loaded) return
        loaded = true
        dir.mkdirs()
        dir.listFiles { f -> f.name.endsWith(".json.tmp") }?.forEach { it.delete() }   // interrupted writes
        dir.listFiles { f -> f.isFile && f.name.endsWith(".json") }?.sortedBy { it.name }?.forEach { f ->
            try {
                val (p, rewrite) = decode(f.readText(Charsets.UTF_8))
                val clash = byName(p.name)
                if (cache.containsKey(p.id) || clash != null || f.name != "${p.id}.json") {
                    // duplicate id/name on disk (e.g. a hand-copied file): keep the first, quarantine the rest
                    if (cache.containsKey(p.id) || clash != null) { quarantine(f); return@forEach }
                    writeAtomic(p); f.delete()
                } else if (rewrite) writeAtomic(p)
                cache[p.id] = p
            } catch (e: InvalidPresetException) {
                if (e.message?.contains("newer than this app") == true) return@forEach   // keep for a future app
                quarantine(f)
            } catch (e: Exception) {
                quarantine(f)
            }
        }
        publish()
    }

    /** Loads from disk (also done lazily by every call). Call once at startup so [presets] emits. */
    suspend fun refresh() = mutex.withLock { ensureLoaded() }

    override suspend fun save(name: String, params: PresetParams, stems: Map<String, JsonObject>?,
                              overwrite: Boolean): FreestylePreset = mutex.withLock {
        ensureLoaded()
        val n = cleanName(name)
        val now = clock()
        val clean = PresetValidator.sanitize(params).params
        val existing = byName(n)
        val p = if (existing != null) {
            if (!overwrite) throw PresetNameConflictException(n, existing.id)
            existing.copy(name = n, params = clean, stems = stems, updatedAt = now)
        } else FreestylePreset(id = newId(), name = n, createdAt = now, updatedAt = now, params = clean, stems = stems)
        writeAtomic(p); cache[p.id] = p; publish(); p
    }

    override suspend fun list() = mutex.withLock { ensureLoaded(); sorted() }
    override suspend fun load(id: String) = mutex.withLock { ensureLoaded(); cache[id] }
    override suspend fun loadByName(name: String) = mutex.withLock { ensureLoaded(); byName(name) }

    override suspend fun rename(id: String, newName: String): FreestylePreset = mutex.withLock {
        ensureLoaded()
        val p = cache[id] ?: throw PresetNotFoundException(id)
        val n = cleanName(newName)
        val other = byName(n)
        if (other != null && other.id != id) throw PresetNameConflictException(n, other.id)
        val r = p.copy(name = n, updatedAt = clock())
        writeAtomic(r); cache[id] = r; publish(); r
    }

    override suspend fun delete(id: String): Boolean = mutex.withLock {
        ensureLoaded()
        val had = cache.remove(id) != null
        fileOf(id).delete()
        if (had) publish()
        had
    }

    override suspend fun exportJson(id: String): String = mutex.withLock {
        ensureLoaded()
        PresetJson.encodeToString(FreestylePreset.serializer(), cache[id] ?: throw PresetNotFoundException(id))
    }

    override suspend fun importJson(json: String, overwrite: Boolean): FreestylePreset = mutex.withLock {
        ensureLoaded()
        var (p, _) = decode(json)
        val now = clock()
        if (cache.containsKey(p.id)) p = p.copy(id = newId())           // never clobber by id
        val clash = byName(p.name)
        if (clash != null) {
            if (overwrite) { cache.remove(clash.id); fileOf(clash.id).delete(); p = p.copy(id = clash.id) }
            else {
                var i = 2
                while (byName("${p.name} ($i)") != null) i++
                p = p.copy(name = cleanName("${p.name.take(58)} ($i)"))
            }
        }
        p = p.copy(updatedAt = now)
        writeAtomic(p); cache[p.id] = p; publish(); p
    }
}

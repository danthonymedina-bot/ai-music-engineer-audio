package com.david.aimusicengineer.presets

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

class PresetRepositoryTest {
    private lateinit var dir: File
    private var now = 1_000L
    private fun repo() = FilePresetRepository(dir, clock = { now++ })

    @Before fun setUp() { dir = Files.createTempDirectory("presets").toFile() }
    @After fun tearDown() { dir.deleteRecursively() }

    private val warm = PresetParams(eq = Eq(Eq.fixedBands(3.0, -1.0, 0.0, 2.0, 4.0)),
        saturation = Saturation(enabled = true, type = SatType.TAPE, driveDb = 9.0, mix = 0.6),
        width = Width(enabled = true, widthPct = 130.0), compressor = Compressor(enabled = true, ratio = 3.0))

    @Test fun saveListLoadSortedAndPersisted() = runBlocking<Unit> {
        val r = repo()
        r.save("zeta", warm); r.save("Alpha", PresetParams()); r.save("beta", warm)
        assertEquals(listOf("Alpha", "beta", "zeta"), r.list().map { it.name })
        assertEquals(listOf("Alpha", "beta", "zeta"), r.presets.first().map { it.name })
        val z = r.loadByName("ZETA")!!
        assertEquals(warm, z.params)
        assertEquals(z, r.load(z.id))
        // a fresh repository over the same dir sees the same data
        val r2 = repo()
        assertEquals(r.list(), r2.list())
        assertTrue(File(dir, "${z.id}.json").exists())
        assertTrue(dir.listFiles()!!.none { it.name.endsWith(".tmp") })
    }

    @Test fun saveNameConflictIsCaseInsensitiveAndOverwriteKeepsId() = runBlocking<Unit> {
        val r = repo()
        val a = r.save("My Vocal", warm)
        val e = assertThrows(PresetNameConflictException::class.java) { runBlocking { r.save("  my   vocal ", PresetParams()) } }
        assertEquals(a.id, e.existingId)
        val b = r.save("MY VOCAL", PresetParams(), overwrite = true)
        assertEquals(a.id, b.id); assertEquals(a.createdAt, b.createdAt); assertTrue(b.updatedAt > a.updatedAt)
        assertEquals(PresetParams(), r.load(a.id)!!.params)
        assertEquals(1, r.list().size)
        assertThrows(InvalidPresetException::class.java) { runBlocking { r.save("   ", warm) } }
    }

    @Test fun renameAndConflicts() = runBlocking<Unit> {
        val r = repo()
        val a = r.save("one", warm); val b = r.save("two", warm)
        assertEquals("uno", r.rename(a.id, "uno").name)
        assertThrows(PresetNameConflictException::class.java) { runBlocking { r.rename(a.id, "TWO") } }
        assertEquals("Uno", r.rename(a.id, "Uno").name)                       // case-only rename of itself is fine
        assertThrows(PresetNotFoundException::class.java) { runBlocking { r.rename("nope", "x") } }
        assertEquals(listOf("two", "Uno"), r.list().map { it.name })
        assertEquals("Uno", repo().load(a.id)!!.name)
        assertNotNull(b)
    }

    @Test fun deleteRemovesFileAndEmits() = runBlocking<Unit> {
        val r = repo()
        val a = r.save("gone", warm)
        assertTrue(r.delete(a.id)); assertFalse(r.delete(a.id))
        assertTrue(r.presets.value.isEmpty()); assertFalse(File(dir, "${a.id}.json").exists())
    }

    @Test fun clampsWithBundledRangesAndFixedBands() = runBlocking<Unit> {
        val r = repo()
        val wild = PresetParams(eq = Eq(listOf(EqBand(freqHz = 5.0, gainDb = 99.0, q = 1e9))),
            saturation = Saturation(driveDb = Double.POSITIVE_INFINITY, mix = -3.0),
            width = Width(widthPct = 1e6), compressor = Compressor(ratio = Double.NaN, attackMs = 0.0))
        val res = PresetValidator.sanitize(wild)
        assertTrue(res.warnings.isNotEmpty())
        val p = r.save("wild", wild).params
        assertEquals(5, p.eq.bands.size)
        assertEquals(30.0, p.eq.gain100, 0.0); assertEquals(18.0, p.eq.bands[0].q, 0.0); assertEquals(100.0, p.eq.bands[0].freqHz, 0.0)
        assertEquals(0.0, p.saturation.driveDb, 0.0)            // non-finite -> default, like the server
        assertEquals(0.0, p.saturation.mix, 0.0); assertEquals(200.0, p.width.widthPct, 0.0)
        assertEquals(4.0, p.compressor.ratio, 0.0); assertEquals(0.1, p.compressor.attackMs, 0.0)
        assertEquals(BandType.HIGHSHELF, p.eq.bands[Eq.AIR_INDEX].type)
    }

    @Test fun exportIsAToneProfileAndImportHandlesConflicts() = runBlocking<Unit> {
        val r = repo()
        val stems = mapOf("vocals" to JsonObject(mapOf("width" to JsonObject(mapOf("widthPct" to JsonPrimitive(80))))))
        val a = r.save("Share me", warm, stems)
        val text = r.exportJson(a.id)
        val o = PresetJson.parseToJsonElement(text).jsonObject
        assertEquals("freestyle", o["mode"]!!.jsonPrimitive.content)
        assertEquals(5, o["params"]!!.jsonObject["eq"]!!.jsonObject["bands"]!!.jsonArray.size)
        assertEquals(1, o["schemaVersion"]!!.jsonPrimitive.content.toInt())
        assertNotNull(o["stems"]!!.jsonObject["vocals"])
        val b = r.importJson(text)                         // same id + name -> new id, "Share me (2)"
        assertNotEquals(a.id, b.id); assertEquals("Share me (2)", b.name); assertEquals(a.params, b.params)
        assertEquals("Share me (3)", r.importJson(text).name)
        val c = r.importJson(text, overwrite = true)       // replaces the "Share me" entry, keeps its id
        assertEquals(a.id, c.id); assertEquals(3, r.list().size)
        assertThrows(InvalidPresetException::class.java) { runBlocking { r.importJson("{oops") } }
        assertThrows(InvalidPresetException::class.java) { runBlocking { r.importJson("""{"schemaVersion":99,"name":"x"}""") } }
    }

    @Test fun corruptFilesAreQuarantinedAndTmpFilesCleaned() = runBlocking<Unit> {
        val r = repo()
        val good = r.save("good", warm)
        File(dir, "11111111-1111-1111-1111-111111111111.json").writeText("{\"name\": \"trunc")
        File(dir, "22222222-2222-2222-2222-222222222222.json").writeText("[1,2,3]")
        File(dir, ".${good.id}.json.tmp").writeText("partial write")
        val r2 = repo()
        assertEquals(listOf("good"), r2.list().map { it.name })
        assertEquals(2, r2.quarantined.value.size)
        assertEquals(2, File(dir, "corrupt").listFiles()!!.size)
        assertFalse(File(dir, ".${good.id}.json.tmp").exists())
        r2.save("after", warm)                               // still writable
        assertEquals(2, repo().list().size)
    }

    @Test fun migratesV0AndRewritesFile() = runBlocking<Unit> {
        val id = "33333333-3333-3333-3333-333333333333"
        File(dir.apply { mkdirs() }, "$id.json").writeText(
            """{"id":"$id","name":"Old","eqGains":[1,2,3,4,50],"drive":6,"widthPct":150}""")
        val p = repo().load(id)!!
        assertEquals(1, p.schemaVersion)
        assertEquals(listOf(1.0, 2.0, 3.0, 4.0, 30.0), p.params.eq.bands.map { it.gainDb })   // 50 clamped
        assertTrue(p.params.saturation.enabled); assertEquals(6.0, p.params.saturation.driveDb, 0.0)
        assertEquals(150.0, p.params.width.widthPct, 0.0)
        val disk = PresetJson.parseToJsonElement(File(dir, "$id.json").readText()).jsonObject
        assertEquals(1, disk["schemaVersion"]!!.jsonPrimitive.content.toInt())
        assertNull(disk["eqGains"])
        // a custom hook runs for each missing version step
        var called = 0
        val custom = mapOf(0 to PresetMigration { o -> called++; PresetMigrations.default.getValue(0).migrate(o) })
        File(dir, "44444444-4444-4444-4444-444444444444.json").writeText("""{"id":"44444444-4444-4444-4444-444444444444","name":"Older"}""")
        FilePresetRepository(dir, migrations = custom).list()
        assertEquals(1, called)
    }

    @Test fun newerSchemaFilesAreKeptNotQuarantined() = runBlocking<Unit> {
        val f = File(dir.apply { mkdirs() }, "55555555-5555-5555-5555-555555555555.json")
        f.writeText("""{"id":"55555555-5555-5555-5555-555555555555","name":"Future","schemaVersion":9}""")
        val r = repo()
        assertTrue(r.list().isEmpty()); assertTrue(f.exists()); assertTrue(r.quarantined.value.isEmpty())
    }

    @Test fun rangesMatchServerSchemaFile() {
        val s = File(System.getProperty("toneSchema") ?: return).readText()
        val o = PresetJson.parseToJsonElement(s).jsonObject["sections"]!!.jsonObject
        val ww = o["width"]!!.jsonObject["widthPct"]!!.jsonObject
        assertEquals(ww["max"]!!.jsonPrimitive.content.toDouble(), ToneRanges.numbers.getValue("width.widthPct").max, 0.0)
        assertEquals(ToneRanges.SCHEMA_VERSION, PresetJson.parseToJsonElement(s).jsonObject["version"]!!.jsonPrimitive.content.toInt())
    }
}

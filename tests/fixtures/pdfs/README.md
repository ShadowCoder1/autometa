# Test fixture PDFs (not distributed)

The tests read three real published papers. They are copyrighted by their publishers, so they are
not committed to this repository; this directory ships empty on purpose.

| file | paper | DOI |
|---|---|---|
| `bock2005.pdf` | Bock (2005), *Components of sensorimotor adaptation in young and elderly subjects* | [10.1007/s00221-004-2147-z](https://doi.org/10.1007/s00221-004-2147-z) |
| `cressman2010.pdf` | Cressman, Salomonczyk & Henriques (2010), *Visuomotor adaptation and proprioceptive recalibration in older adults* | [10.1007/s00221-010-2381-5](https://doi.org/10.1007/s00221-010-2381-5) |
| `heuer2008.pdf` | Heuer & Hegele (2008), *Adaptation to visuomotor rotations in younger and older adults* | [10.1037/a0013283](https://doi.org/10.1037/a0013283) |

Download each from its DOI (through your institution, or an author copy) and save it here under
the file name in the table. `tests/conftest.py` expects `bock2005.pdf`; the other two are used by
the digitiser and figure-ingest tests.

Any PDF of the right paper works — the tests key on the document's content, not on a checksum.
Tests that need a fixture will fail with a missing-file error until you supply it; the rest of the
suite runs without them.

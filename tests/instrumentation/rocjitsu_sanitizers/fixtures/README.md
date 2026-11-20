# Waitcheck reporting fixtures

Minimal text derivatives of the retained 11 November 2026 cap64/cap1 control
logs. The object path alone is replaced by `/fixtures/multiple_missing_waits`;
the summary numbers, gfx1250/image0/entry0, diagnostic offsets/instructions and
producer/consumer text are retained. No binary or private path is included.
Whitespace is normalized to LF. Counts/observations were previously established
by the saved `inspect_partial.py` analyzer. The updated saved reader additionally
recognizes the cap1 terminal `omitted 2 diagnostic(s) after limit`; retain it.
The old reader's count gate had failed before considering this line. A synthetic
test separately removes the signal to verify that bare mismatches stay unknown.
Fixtures are parser data, never run.

The tests substitute the portable object path with an inert temporary file when
checking recipe/backend integration. Native4 is supplied by a fake executor.
These tests do not establish original binary provenance or native qualification.

"""Private local evaluation history. No network, automatic export or model training."""
from contextlib import closing
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3

from .pipeline import Execution, Step


def default_path():
    if os.name == 'nt':
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        root = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state'))
    return root / 'superwhisper-custom' / 'history.sqlite3'


class HistoryStore:
    def __init__(self, path=None):
        self.path = Path(path or default_path()).expanduser().resolve()
        if any((parent / '.git').is_file() or (parent / '.git' / 'HEAD').exists()
               for parent in self.path.parents):
            raise ValueError('History must be outside any Git repository')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != 'nt':
            self.path.parent.chmod(0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        if os.name != 'nt':
            self.path.chmod(0o600)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at)')
            db.execute('CREATE TABLE IF NOT EXISTS audio (id TEXT PRIMARY KEY, sample_rate INTEGER NOT NULL, samples BLOB NOT NULL)')
            if 'codec' not in {row[1] for row in db.execute('PRAGMA table_info(audio)')}:
                db.execute("ALTER TABLE audio ADD COLUMN codec TEXT NOT NULL DEFAULT 'lossless'")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=0.25)
        db.execute('PRAGMA secure_delete=ON')
        return db

    def save(self, report, audio=None, sample_rate=16000, codec="lossless"):
        compressed = None
        if audio is not None:
            import numpy as np
            samples = np.asarray(audio, dtype='<f4')
            if samples.ndim != 1:
                raise ValueError('Mono audio expected')
            from .audio_archive import encode
            try:
                compressed = encode(samples, sample_rate, codec)
                report.metrics['audio_storage'] = {
                    'saved': True, 'encoding': codec, 'lossless': codec == 'lossless',
                    'target_bitrate_bps': 32000 if codec == 'opus' else None, 'sample_rate': sample_rate,
                    'samples': len(samples), 'uncompressed_bytes': samples.nbytes,
                    'compressed_bytes': len(compressed),
                }
            except Exception:
                report.metrics['audio_storage'] = {'saved': False, 'encoding': codec,
                                                    'error': 'Audio encoding unavailable'}
        payload = json.dumps(asdict(report), ensure_ascii=False, allow_nan=False)
        with closing(self._connect()) as db, db:
            # Re-delivery must neither duplicate a run nor erase its manual review.
            db.execute('INSERT OR IGNORE INTO runs VALUES (?, ?, ?)',
                       (report.run_id, report.created_at, payload))
            if compressed is not None:
                db.execute('INSERT OR IGNORE INTO audio VALUES (?, ?, ?, ?)',
                           (report.run_id, sample_rate, compressed, codec))

    def recent(self, limit=200):
        with closing(self._connect()) as db:
            rows = db.execute('SELECT payload FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?', (limit,)).fetchall()
        reports = []
        for (payload,) in rows:
            data = json.loads(payload)
            data['steps'] = [Step(**step) for step in data['steps']]
            reports.append(Execution(**data))
        return reports

    def review(self, run_id, rating, correction, reference_transcript=None):
        if rating not in ('unrated', 'good', 'incorrect'):
            raise ValueError('Unknown rating')
        with closing(self._connect()) as db, db:
            row = db.execute('SELECT payload FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                return False
            data = json.loads(row[0])
            previous_reference = data.get('review', {}).get('reference_transcript')
            data['review'] = {'rating': rating, 'correction': correction}
            if reference_transcript is not None or previous_reference is not None:
                data['review']['reference_transcript'] = (reference_transcript if reference_transcript is not None
                                                         else previous_reference)
            db.execute('UPDATE runs SET payload=? WHERE id=?',
                       (json.dumps(data, ensure_ascii=False), run_id))
        return True

    def audio(self, run_id):
        with closing(self._connect()) as db:
            row = db.execute('SELECT sample_rate, samples, codec FROM audio WHERE id=?', (run_id,)).fetchone()
        if row is None:
            return None
        from .audio_archive import decode
        return decode(row[1], row[0], row[2]), row[0]

    def delete(self, run_id):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM audio WHERE id=?', (run_id,))
            db.execute('DELETE FROM runs WHERE id=?', (run_id,))

    def clear(self):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM audio')
            db.execute('DELETE FROM runs')
        # SQLite secure_delete overwrites deleted cell contents. This is not a
        # guarantee of forensic erasure on SSDs, backups or filesystem snapshots.

    def summary(self):
        with closing(self._connect()) as db:
            return db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]

    def statistics(self):
        import math
        from collections import Counter
        from statistics import median
        timings = {key: [] for key in ('pipeline_duration_ms', 'transcription_total_ms', 'stop_to_result_ms')}
        ratings = Counter()
        warnings = count = audio_runs = audio_bytes = input_tokens = output_tokens = 0
        device_samples = 0
        with closing(self._connect()) as db:
            for (payload,) in db.execute('SELECT payload FROM runs'):
                data = json.loads(payload)
                count += 1
                warnings += bool(data.get('warning'))
                ratings[data.get('review', {}).get('rating', 'unrated')] += 1
                measurements = data.get('metrics', {})
                audio_storage = measurements.get('audio_storage', {})
                if audio_storage.get('saved'):
                    audio_runs += 1
                audio_bytes += audio_storage.get('compressed_bytes', 0)
                input_tokens += measurements.get('input_tokens', 0) or 0
                output_tokens += measurements.get('output_tokens', 0) or 0
                device_samples += measurements.get('samples', 0) or 0
                for key, values in timings.items():
                    value = measurements.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                        values.append(value)
        stats = {'runs': count, 'runs_with_warnings': warnings,
                 'manual_ratings': dict(ratings), 'compressed_audio_bytes': audio_bytes,
                 'audio_runs': audio_runs, 'input_tokens': input_tokens,
                 'output_tokens': output_tokens, 'device_samples': device_samples,
                 'database_bytes': self.path.stat().st_size}
        for key, values in timings.items():
            if values:
                stats[key] = {'measured_runs': len(values), 'median': median(values),
                              'p95': sorted(values)[math.ceil(.95 * len(values)) - 1]}
        return stats


if __name__ == '__main__':
    print(json.dumps(HistoryStore().statistics(), indent=2, ensure_ascii=False))

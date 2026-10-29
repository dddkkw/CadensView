"""Resume an isolated color rebuild, then publish one source in a transaction."""
from dataclasses import asdict
import shutil
import time
import zlib

from .cache import ColorCache, default_cache_directory
from .colors import ColorParameters, analyze_colors, check_cancel
from .color_summary import SummaryParameters
from .model import digest


def staging_directory(context, cache_dir, parameters):
    from pathlib import Path
    identity = digest([context.source_identity['key'], parameters.method(), asdict(SummaryParameters())])
    return (Path(cache_dir) if cache_dir is not None else default_cache_directory()) / 'rebuild' / identity


def analyze_or_rebuild(context, cache_dir, parameters=ColorParameters(), *, rebuild=False,
                       cancel=None, progress=None, cooperate=None):
    stage = staging_directory(context, cache_dir, parameters)
    active = stage / 'active'
    if not rebuild and not active.exists():
        return analyze_colors(context, cache_dir, parameters, cancel, progress, cooperate, include_summary=True)
    check_cancel(cancel)
    stage.mkdir(parents=True, exist_ok=True)
    if not active.exists():
        # Only this source/method's disposable staging database; published cache stays intact.
        with ColorCache(stage) as cache:
            with cache.connection:
                cache.connection.execute('DELETE FROM features')
                cache.connection.execute('DELETE FROM results')
        active.touch()
    result = analyze_colors(context, stage, parameters, cancel, progress, cooperate, include_summary=True)
    if any(result[k]['status'] == 'unsupported' for k in ('video_color_timeline','shot_color_statistics')):
        raise ValueError('此源不支持完整色彩重建，原缓存保留')
    check_cancel(cancel)
    source, method = context.source_identity['key'], parameters.feature_key()
    # New image URLs keep Qt's image cache from showing an older still after replacement.
    suffix = f'-r{time.time_ns()}'
    with ColorCache(stage) as pending, ColorCache(cache_dir) as target:
        previews = target.directory / 'previews'
        previews.mkdir(exist_ok=True)
        names = {}
        for still in result['video_color_summary']['representatives']:
            check_cancel(cancel)
            old = still['file']
            new = old.removesuffix('.png') + suffix + '.png'
            temporary = previews / (new + '.tmp')
            shutil.copyfile(stage / 'previews' / old, temporary)
            temporary.replace(previews / new)
            names[old] = new
            still['file'] = new
        rows = []
        for key, payload in pending.connection.execute('SELECT key,payload FROM results'):
            value = pending.decode(payload)
            if 'representatives' in value:
                for still in value['representatives']:
                    still['file'] = names[still['file']]
            rows.append((key, target.encode(value)))
        db = target.connection
        with db:
            # Includes legacy single-frame palette keys, whose payload lacked source metadata.
            frames = [n for n, in db.execute('SELECT frame FROM features WHERE source=? AND method=?', (source,method))]
            stale = {digest([source,method,n,parameters.palette_colors,parameters.palette_iterations,'frame-palette-v1']) for n in frames}
            for key,payload in db.execute('SELECT key,payload FROM results'):
                try:
                    value = target.decode(payload)
                    if value.get('metadata',{}).get('source_identity',{}).get('key') == source:
                        stale.add(key)
                except (ValueError, zlib.error):
                    pass  # A corrupt matching key is replaced by the new validated group below.
            db.executemany('DELETE FROM results WHERE key=?', ((k,) for k in stale))
            db.execute('DELETE FROM features WHERE source=? AND method=?', (source,method))
            features = pending.connection.execute('SELECT * FROM features WHERE source=? AND method=?', (source,method))
            db.executemany('INSERT OR REPLACE INTO features VALUES (?,?,?,?)', features)
            db.executemany('INSERT OR REPLACE INTO results VALUES (?,?)', rows)
            check_cancel(cancel)
        active.unlink(missing_ok=True)
    result['performance']['rebuilt'] = True
    return result

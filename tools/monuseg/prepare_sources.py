"""Prepare original MoNuSeg TIFF/XML files for the recorded reconstruction."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path, help='Original training ZIP')
    parser.add_argument('--test', required=True, type=Path, help='Original test ZIP')
    parser.add_argument('--output', required=True, type=Path, help='Extracted reconstruction directory')
    args = parser.parse_args()
    records = Path(__file__).resolve().parent
    manifest = json.loads((records / 'manifest.json').read_text())
    verification = json.loads((records / 'original_archive_verification.json').read_text())
    for split, archive in [('train', args.train), ('test', args.test)]:
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != verification[split]['original_archive_sha256']:
            raise ValueError('Original {} archive checksum does not match.'.format(split))
        patients = manifest['patients'][split]
        if split == 'train':
            patients = patients + manifest['patients']['val']
        destination = args.output / 'source' / split
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as source:
            names = [name for name in source.namelist()
                     if '__MACOSX' not in Path(name).parts and not name.endswith('/')]
            for patient in sorted(patients):
                for kind, extensions in [('xml', {'.xml'}), ('png', {'.tif', '.tiff'})]:
                    matches = [name for name in names if Path(name).stem == patient
                               and Path(name).suffix.lower() in extensions]
                    if len(matches) != 1:
                        raise ValueError('Expected one {} for {}'.format(kind, patient))
                    content = source.read(matches[0])
                    target = destination / (patient + '.' + kind)
                    if kind == 'png':
                        with Image.open(io.BytesIO(content)) as image:
                            if image.size != (1000, 1000):
                                raise ValueError('Unexpected source dimensions: ' + patient)
                            image.convert('RGB').save(target)
                    else:
                        target.write_bytes(content)
        print('{}: prepared {} source cases'.format(split, len(patients)))
    dataset = args.output / 'monuseg'
    dataset.mkdir(parents=True, exist_ok=True)
    for name in ['manifest.json', 'original_archive_verification.json']:
        shutil.copyfile(records / name, dataset / name)


if __name__ == '__main__':
    main()

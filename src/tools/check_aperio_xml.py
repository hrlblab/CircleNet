"""Check annotation XML structure/coordinates without an OpenSlide install.

This does not validate ImageScope rendering or vendor-specific SCN transforms.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET


def check_annotations(path, width=None, height=None):
    root = ET.parse(path).getroot()
    regions = [node for node in root.iter() if node.tag.split('}')[-1] == 'Region']
    errors, warnings, coordinates = [], [], []
    if not regions:
        errors.append('No Region elements: there are no annotations to display.')
    for index, region in enumerate(regions):
        vertices = [node for node in region.iter() if node.tag.split('}')[-1] == 'Vertex']
        if len(vertices) < 2:
            errors.append('Region {} has fewer than two vertices.'.format(index + 1))
        for vertex in vertices:
            try:
                x, y = float(vertex.attrib['X']), float(vertex.attrib['Y'])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError('non-finite coordinate')
                coordinates.append((x, y))
            except (KeyError, ValueError):
                errors.append('Region {} contains an invalid X/Y coordinate.'.format(index + 1))
    if coordinates:
        if any(x < 0 or y < 0 or (width is not None and x > width)
               or (height is not None and y > height) for x, y in coordinates):
            warnings.append('Coordinates extend outside the supplied level-0 bounds; check crop offsets, downsampling, and vendor transforms.')
    return dict(regions=len(regions), vertices=len(coordinates), errors=errors,
                warnings=warnings, coordinate_bounds=([min(x for x, y in coordinates),
                min(y for x, y in coordinates), max(x for x, y in coordinates),
                max(y for x, y in coordinates)] if coordinates else None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml')
    parser.add_argument('--width', type=float, help='Level-0 slide width in pixels')
    parser.add_argument('--height', type=float, help='Level-0 slide height in pixels')
    args = parser.parse_args()
    try:
        report = check_annotations(args.xml, args.width, args.height)
    except (OSError, ET.ParseError) as exc:
        parser.exit(1, '{}\n'.format(exc))
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['errors'] else 0)

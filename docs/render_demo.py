"""Regenerate the README's scientific animation from synthetic geometry."""
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from cbct_width.phantoms import make_phantom
from cbct_width.pipeline import measure_widths


def main():
    output = Path(__file__).resolve().parent / 'synthetic-demo.gif'
    image = make_phantom()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'phantom.nii.gz'
        nib.save(image, path)
        result = measure_widths(path)
    fig = plt.figure(figsize=(10, 5.2), facecolor='#f6f9fc')
    ax = fig.add_subplot(111, projection='3d', facecolor='#f6f9fc')
    data = np.asanyarray(image.dataobj)
    for label, color in ((14, '#127a8b'), (6, '#39a5b2'), (22, '#3567aa'), (30, '#71a0df')):
        vertices, faces, _, _ = marching_cubes((data == label).astype(float), 0.5, step_size=2)
        points = nib.affines.apply_affine(image.affine, vertices)
        ax.add_collection3d(Poly3DCollection(points[faces], facecolor=color, edgecolor='none', alpha=0.88))
    for a, b in ((14, 6), (22, 30)):
        points = np.stack([result.per_tooth[a].furcation_mm, result.per_tooth[b].furcation_mm])
        ax.plot(*points.T, color='#c7442c', linewidth=2.5)
        ax.scatter(*points.T, color='#f5bd48', edgecolors='#553510', s=45, depthshade=False)
    ax.set(xlim=(0, 72), ylim=(0, 52), zlim=(0, 40), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
    ax.set_box_aspect((72, 52, 40))
    ax.tick_params(labelsize=8)
    fig.suptitle('Synthetic four-molar geometry', fontsize=19, fontweight='bold', color='#20364c', y=0.96)
    fig.text(0.5, 0.875, f'Maxillary width {result.maxilla_width_mm:.2f} mm  ·  Mandibular width {result.mandible_width_mm:.2f} mm',
             ha='center', fontsize=12, color='#20364c')
    fig.text(0.5, 0.035, 'Measured by cbct-width  •  Toy shapes, no patient data  •  Not clinical validation',
             ha='center', fontsize=10, color='#4d6578')
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.03, right=0.98)
    def frame(angle):
        ax.view_init(elev=23, azim=angle)
        return []
    animation = FuncAnimation(fig, frame, frames=np.arange(-70, 290, 20), interval=160)
    animation.save(output, writer=PillowWriter(fps=6), dpi=100)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()

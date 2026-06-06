import posterCover from '../assets/poster.png';
import overviewDiagram from '../assets/overview_diagram.png';

// All page content lives here so the site is easy to extend.
// To add more next week, edit the arrays below (and drop new images in ../assets).
export const project = {
  title: 'SINCPS',
  subtitle: 'Semantic-aware Implicit Neural Compression for Physics Simulations',
  venue: 'PASC26 — ACM Student Research Competition — University of Bern, Switzerland — June 2026',
  authors: [
    { name: 'Jessica Ezemba', affil: 'Carnegie Mellon University' },
    { name: 'James Afful', affil: 'Iowa State University' },
    { name: 'Mei-Yu Wang', affil: 'Pittsburgh Supercomputing Center' },
  ],
  tags: [
    'Implicit Neural Representations',
    'Scientific Data Compression',
    'SIREN',
    'Physics Simulations',
    'Wafer-Scale Computing',
    'The Well',
  ],
  skills: ['Python', 'PyTorch', 'Cerebras CS-3', 'SIREN', 'Fourier Features'],
  links: [
    { label: 'Code and models', url: 'https://github.com/Jezemba/SINCPS' },
    { label: 'The Well benchmark', url: 'https://github.com/PolymathicAI/the_well' },
  ],
  coverImage: posterCover,

  summary: `Machine learning surrogates and data-driven scientific discovery need efficient access to simulation data, yet physics simulations generate terabyte-scale datasets. Traditional compression either achieves insufficient ratios or corrupts physics-critical features like conservation laws. Implicit neural representations offer a promising alternative, but adoption has been limited by lengthy training times and dataset-specific fitting. SINCPS uses wafer-scale computing to train one model per dataset in 2 to 3 hours. Across 22 datasets from The Well benchmark, it achieves 150x to 25,000x compression. Turbulent flows and 3D data remain challenging, but about half of the datasets exceed 20 dB PSNR, enabling large simulation archives to enter discovery workflows.`,

  technical: `The pipeline turns HDF5 simulation files into coordinate-value pairs. Static fields are stored as float16, and dynamic fields are network-encoded with z-score normalization to balance disparate physical magnitudes. Multiple coordinate systems are supported, including Cartesian, spherical, and log-spherical grids. The model is a SIREN network with Fourier positional encoding: input coordinates pass through 10 levels of Fourier features with frequencies from pi to 512 pi, then four hidden layers of 1024 units with sinusoidal activations, mapping spatiotemporal coordinates to physical field values. The network has about 4.2 million parameters, and each saved model is 37.6 MB. Training uses MSE loss with Adam (learning rate 1e-4 with cosine decay), 50,000 steps, and batch size 16,384, on the Cerebras CS-3.`,

  contributions: [
    'A compression framework achieving 150x to 25,000x ratios across diverse physics domains.',
    'Physics-aware validation that checks conservation-law preservation, not just pixel-level accuracy.',
    'A demonstration that wafer-scale computing makes implicit neural compression practical at benchmark scale.',
  ],

  results: [
    '150x to 25,000x compression across 22 datasets from The Well benchmark.',
    'Each model compresses to 37.6 MB, from initial sizes of 5.4 GB to 1.9 TB.',
    'Training is 2 to 3 hours per dataset on the Cerebras CS-3, compared with 6.6 hours on an NVIDIA H100 GPU and 11.2 hours on an AMD EPYC 7702P CPU.',
    'Smooth fields such as astrophysics reach about 32 dB PSNR; turbulent and 3D fields are harder at about 13 dB.',
    'About half of the datasets exceed 20 dB PSNR.',
  ],

  validation: [
    'Spectral error, the relative FFT error between reconstruction and ground truth, with an acceptable tolerance under 15 percent.',
    'Relative L2 error acceptable under 10 percent.',
    'Conservation-law errors held to a strict 1 percent primary tolerance, with domain-specific validators (for example mass, energy, and divergence of B).',
  ],

  images: [
    { src: overviewDiagram, caption: 'SINCPS overview: simulation data, train neural function, store neural representation, query at arbitrary resolution.' },
  ],

  acknowledgments: `This work was supported by the ByteBoost cybertraining program (NSF awards 2320990, 2320991, 2320992), the Neocortex project (NSF 2005597), the ACES platform (NSF 2112356), and the Ookami cluster (NSF 1927880), with technical support from Cerebras for the CS-3.`,
};

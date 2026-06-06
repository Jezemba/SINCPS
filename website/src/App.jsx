import { FiGithub, FiExternalLink } from 'react-icons/fi';
import { project } from './data/project';
import { FadeIn, StaggerContainer, StaggerItem } from './components/Motion';

function SectionHeading({ children }) {
  return <h2 className="text-xl font-bold text-gray-900 mb-3">{children}</h2>;
}

function BulletList({ items, color }) {
  return (
    <StaggerContainer className="space-y-2">
      {items.map((item, i) => (
        <StaggerItem key={i}>
          <div className="flex items-start gap-3">
            <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${color}`} />
            <span className="text-gray-600 leading-relaxed">{item}</span>
          </div>
        </StaggerItem>
      ))}
    </StaggerContainer>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-white text-gray-900 antialiased">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Header */}
        <FadeIn>
          <p className="text-xs font-semibold text-blue-600 uppercase tracking-wider mb-2">
            {project.venue}
          </p>
          <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-gray-900 leading-tight">
            {project.title}
          </h1>
          <p className="mt-2 text-lg text-gray-500">{project.subtitle}</p>

          {/* Authors */}
          <p className="mt-4 text-sm text-gray-500">
            {project.authors.map((a, i) => (
              <span key={a.name}>
                <span className="text-gray-700 font-medium">{a.name}</span>
                <span className="text-gray-400"> ({a.affil})</span>
                {i < project.authors.length - 1 ? ', ' : ''}
              </span>
            ))}
          </p>

          {/* Tags + Skills */}
          <div className="mt-4 flex flex-wrap gap-2">
            {[...project.tags, ...project.skills.filter((s) => !project.tags.includes(s))].map((tag) => (
              <span key={tag} className="px-2.5 py-0.5 text-xs font-medium text-gray-500 bg-gray-100 rounded-full">
                {tag}
              </span>
            ))}
          </div>

          {/* Links */}
          {project.links.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {project.links.map((link) => (
                <a
                  key={link.label}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  {link.url.includes('github.com') ? <FiGithub className="w-3.5 h-3.5" /> : <FiExternalLink className="w-3.5 h-3.5" />}
                  {link.label}
                </a>
              ))}
            </div>
          )}
        </FadeIn>

        {/* Cover image */}
        <FadeIn delay={0.05}>
          <div className="mt-8 rounded-xl overflow-hidden border border-gray-100 shadow-sm bg-gray-50">
            <img src={project.coverImage} alt={`${project.title} poster`} className="w-full object-contain max-h-[36rem] mx-auto" />
          </div>
        </FadeIn>

        {/* Overview */}
        <FadeIn className="mt-12">
          <SectionHeading>Overview</SectionHeading>
          <p className="text-gray-600 leading-relaxed">{project.summary}</p>
        </FadeIn>

        {/* Technical Approach */}
        <FadeIn className="mt-10">
          <SectionHeading>Technical Approach</SectionHeading>
          <p className="text-gray-600 leading-relaxed">{project.technical}</p>
        </FadeIn>

        {/* Key Contributions */}
        <FadeIn className="mt-10">
          <SectionHeading>Key Contributions</SectionHeading>
          <BulletList items={project.contributions} color="bg-blue-400" />
        </FadeIn>

        {/* Results */}
        <FadeIn className="mt-10">
          <SectionHeading>Results</SectionHeading>
          <BulletList items={project.results} color="bg-emerald-400" />
        </FadeIn>

        {/* Validation */}
        <FadeIn className="mt-10">
          <SectionHeading>Validation</SectionHeading>
          <BulletList items={project.validation} color="bg-violet-400" />
        </FadeIn>

        {/* Figures */}
        {project.images && project.images.length > 0 && (
          <FadeIn className="mt-12">
            <SectionHeading>Figures</SectionHeading>
            <div className="space-y-8">
              {project.images.map((img, i) => (
                <figure key={i}>
                  <div className="rounded-xl overflow-hidden border border-gray-100 shadow-sm bg-gray-50">
                    <img src={img.src} alt={img.caption} className="w-full object-contain max-h-96" />
                  </div>
                  {img.caption && (
                    <figcaption className="mt-2 text-sm text-gray-400 text-center italic">
                      {img.caption}
                    </figcaption>
                  )}
                </figure>
              ))}
            </div>
          </FadeIn>
        )}

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-100">
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Acknowledgments</h2>
          <p className="text-sm text-gray-400 leading-relaxed">{project.acknowledgments}</p>
          <div className="mt-6 flex flex-wrap gap-4">
            {project.links.map((link) => (
              <a key={link.label} href={link.url} target="_blank" rel="noopener noreferrer"
                 className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-blue-600 transition-colors">
                {link.url.includes('github.com') ? <FiGithub className="w-4 h-4" /> : <FiExternalLink className="w-4 h-4" />}
                {link.label}
              </a>
            ))}
          </div>
        </footer>
      </div>
    </div>
  );
}

import type { Blog } from "../../input_model"

function parseFrontmatter(raw: string): Record<string, string> {
    const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/)
    if (!match) return {}
    const result: Record<string, string> = {}
    for (const line of match[1].split('\n')) {
        const colon = line.indexOf(':')
        if (colon === -1) continue
        const key = line.slice(0, colon).trim()
        const value = line.slice(colon + 1).trim()
        result[key] = value
    }
    return result
}

/** @type {import('./$types').PageLoad} */
export async function load() {
    const rawFiles = import.meta.glob('../../lib/assets/blogs/*.md', { eager: true, query: '?raw', import: 'default' })

    let blogs: Blog[] = []

    for (const path in rawFiles) {
        const slug = path.split('/').at(-1)?.replace('.md', '')
        if (!slug) continue

        const meta = parseFrontmatter(rawFiles[path] as string)
        if (meta.stage !== 'live') continue

        blogs.push({
            title: meta.title,
            image: meta.image ?? null,
            description: meta.description,
            date: new Date(meta.date),
            id: slug,
            stage: meta.stage,
            link: meta.link
        })
    }

    blogs.sort((a, b) => b.date.getTime() - a.date.getTime())
    return { blogs: blogs.slice(0, 3) }
}

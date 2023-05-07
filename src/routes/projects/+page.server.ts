
export async function load(url: string) {
    const paths = import.meta.glob('/src/lib/assets/projects/*.md',
        { eager: true })

    let projects: any[] = []
    for (const path in paths) {
        const file: any = paths[path]
        const slug = path.split('/').at(-1)?.replace(".md", "")
        const metadata = file.metadata
        const project = { ...metadata, slug }
        projects.push(project)


    }
    return { projects }
}
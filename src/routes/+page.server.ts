import type { Blog } from "../input_model"

/** @type {import('./$types').PageLoad} */
export async function load({ params }) {

    let blogs: Blog[] = []

    const paths = import.meta.glob('./blogs/[blog_id]/blogs/*.md', { eager: true })

    console.log(paths)
    for (const path in paths) {
        const file = paths[path]
        const slug = path.split('/').at(-1)?.replace('.md', '')

        if (file && typeof file === 'object' && 'metadata' in file && slug) {
            const metadata = file.metadata as Omit<Blog, 'slug'>
            const blog: Blog = {
                title: metadata.title,
                image: metadata.image,
                description: metadata.description,
                date: metadata.date,
                id: slug,
                stage: metadata.stage,


            }
            if(blog.stage == "live"){
                blogs.push(blog)
            }
           
        }
    }
    return { blogs }

}
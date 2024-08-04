
/** @type {import('./$types').PageLoad} */

export async function load({ params }) {
    const blog = await import(`../../../../lib/assets/blogs/${params.blog_id}.md`);

    return {
        id: params.blog_id,
        title: blog.metadata.title,
        html: blog.default.render().html,
        image_url: blog.metadata.image,
        description: blog.metadata.description,
        date:new Date( blog.metadata.date),
    }

}
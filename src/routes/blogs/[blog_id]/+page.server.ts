
/** @type {import('./$types').PageLoad} */
export async function load({ params }) {

    const blog = await import(`../[blog_id]/blogs/${params.blog_id}.md`);

    console.log(blog.default.render())

    return{
        id: params.blog_id,
        title: blog.metadata.title,
        html: blog.default.render().html
    }

}
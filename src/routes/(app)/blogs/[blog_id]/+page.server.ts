import { error } from '@sveltejs/kit';

/** @type {import('./$types').PageLoad} */

export async function load({ params }) {
    if (!/^[a-zA-Z0-9_-]+$/.test(params.blog_id)) {
        throw error(404, 'Not found');
    }
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
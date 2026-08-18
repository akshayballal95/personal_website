import { error } from '@sveltejs/kit';
import { render } from 'svelte/server';

/** @type {import('./$types').PageLoad} */

export async function load({ params }) {
    if (!/^[a-zA-Z0-9_-]+$/.test(params.blog_id)) {
        error(404, 'Not found');
    }
    const blog = await import(`../../../../lib/assets/blogs/${params.blog_id}.md`);

    return {
        id: params.blog_id,
        title: blog.metadata.title,
        html: render(blog.default).body,
        image_url: blog.metadata.image,
        description: blog.metadata.description,
        date:new Date( blog.metadata.date),
    }

}
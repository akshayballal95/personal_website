import { redirect } from '@sveltejs/kit';

export function load() {
	throw redirect(308, 'https://hashnode-blog-sigma.vercel.app/blog');
}

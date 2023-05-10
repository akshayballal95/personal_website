// import adapter from '@sveltejs/adapter-auto';
import adapter from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/kit/vite';

import addClasses from "rehype-add-classes"
import path from 'path'
import shiki from 'shiki'
/** @type {import('@sveltejs/kit').Config} */
import { escapeSvelte, mdsvex } from 'mdsvex'
import { fileURLToPath } from 'url';

const dirName = path.resolve(fileURLToPath(import.meta.url), '../');

/** @type {import('mdsvex').MdsvexOptions} */
const mdsvexOptions = {
	// layout: {blog:path.join(dirName, './src/routes/blogs/[blog_id]/_layout.svelte')},
	extensions: ['.md', '.svx'],
	rehypePlugins: [[addClasses, {"img":"rounded-md"}]],
	highlight: {
		highlighter: async (code, lang = "text") => {
			const highlighter = await shiki.getHighlighter({ theme: "poimandres" });
			const html = escapeSvelte(highlighter.codeToHtml(code, {lang}));
			return html
		
	}
}}

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex(mdsvexOptions)],
	extensions: ['.svelte', '.md', '.svx'],




	kit: {
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter()
	}
};

export default config;
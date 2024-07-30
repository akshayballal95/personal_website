// import adapter from '@sveltejs/adapter-auto';
import adapter from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/kit/vite';

import addClasses from "rehype-add-classes"
import path from 'path'
import shiki from 'shiki'
/** @type {import('@sveltejs/kit').Config} */
import { escapeSvelte, mdsvex } from 'mdsvex'
import { fileURLToPath } from 'url';
import rehypeKatexSvelte from "rehype-katex-svelte";
import remarkMath from 'remark-math'
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeExternalLinks from 'rehype-external-links';


const dirName = path.resolve(fileURLToPath(import.meta.url), '../');

/** @type {import('mdsvex').MdsvexOptions} */
const mdsvexOptions = {
	extensions: ['.md', '.svx', '.mdx'],
	remarkPlugins: [
		remarkMath,
	  ],
	rehypePlugins: [[addClasses, {"img":"rounded-md", "ul":"list-disc list-inside ml-5 ", "ol":"list-outside", "p":"text-base" }], rehypeKatexSvelte,

	  rehypeExternalLinks,
	[
		rehypeAutolinkHeadings,
		{
			// Adds hyperlinks to the headings, requires rehypeSlug
			behavior: 'prepend',
			properties: { className: ['heading-link'], title: 'Permalink', ariaHidden: 'true' },
			content: {
				type: 'element',
				tagName: 'span',
				properties: {},
				children: [{ type: 'text', value: '#' }]
			}
		}
	]
],
	highlight: {
		highlighter: async (code, lang = "text") => {
			const highlighter = await shiki.getHighlighter({ theme: "github-dark" });
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
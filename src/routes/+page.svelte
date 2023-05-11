<script lang="ts">
	import Aboutme from '$lib/aboutme/aboutme.svelte';
	import BlogList from '$lib/blogs/blog_list.svelte';
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';
	import type { Blog } from '../input_model';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };
	onMount(() => {
		$tabSet = 0;
	});
</script>

<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->

<svelte:head>
	<title>About Me</title>
	<meta
		name="description"
		content="I am passionate about building new technologies that push the boundaries of the status quo.
	My mix of knowledge in both mechanical systems and software development allows me to build
	more holistic systems and products. I have experience in building additive manufacturing
	machines and technologies and have put together several machines from scratch. I am the
	co-inventor of 7 granted patents in the field of composites and additive manufacturing."
	/>
</svelte:head>

<div class="h-full flex items-center p-3">
	<div class="flex xl:flex-row flex-col gap-8 justify-around">
		<div >
			<Aboutme />
		</div>

		<div class="flex flex-col gap-3 lg:basis-2/5 self-start">
			<h3>Recent Blogs</h3>
			<BlogList blogs={data.blogs} />
		</div>
	</div>
</div>

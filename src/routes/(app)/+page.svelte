<script lang="ts">
	import Aboutme from '$lib/aboutme/aboutme.svelte';
	import BlogList from '$lib/blogs/blog_list.svelte';
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';
	import type { Blog } from '../../input_model';

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
		content="Welcome to my personal website! I am a self-taught AI developer driven by a passion for pushing the boundaries of technology. Applying First Principles thinking, I strive to solve complex challenges and create innovative solutions. As a Technology Enthusiast, I constantly explore the latest advancements in the field. I am deeply committed to leveraging AI for social good and advocate for green technology. Join me on this journey as I utilize my self-taught expertise to build products and technologies that address industrial problems. Let's collaborate and shape a better future together."
	/>
	<meta name="title" content="Akshay Ballal: Machine Learning Enthusiast">
</svelte:head>

<div class="h-full flex justify-center items-center p-3 xl:p-12 ">
	<div class="flex xl:flex-row flex-col gap-10 w-full max-w-7xl">
		<div class="xl:basis-3/5 xl:max-w-3xl" >
			<Aboutme />
		</div>

		<div class="flex flex-col gap-3 xl:basis-2/5">
			<h3>Recent Blogs</h3>
			<BlogList blogs={data.blogs} />
		</div>
	</div>
</div>

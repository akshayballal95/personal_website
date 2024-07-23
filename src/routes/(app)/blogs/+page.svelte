<script lang="ts">
	import { onMount } from 'svelte';
	import type { Blog } from '../../../input_model';
	import { tabSet } from '$lib/stores/stateStore';
	import BlogList from '$lib/blogs/blog_list.svelte';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };

	onMount(() => {
		$tabSet = 4;
	});
</script>

<svelte:head>
	<title>My Blogs</title>
	<meta name="description" content="Blogs written by Akshay Ballal" />
	<meta name="keywords" content="Artificial Intelligence, Machine Learning, Programming, Rust" />
</svelte:head>

<div class="lg:flex-row flex flex-col w-full items-start justify-around p-4  ">
	<div class="flex flex-col lg:w-2/4 gap-10"><h1> My Blogs </h1>

		<div  >
			<BlogList blogs={data.blogs} />
		</div></div>
	
	<a href="https://app.daily.dev/akshayballal"><img src="https://api.daily.dev/devcards/c918e620012843b1acae4980c12a7369.png?r=32f" class="lg:w-80" alt="Akshay Ballal's Dev Card"/></a>
	
</div>

<script lang="ts">
	import { onMount } from 'svelte';
	import type { Blog } from '..//../input_model';
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

<div class="flex w-full items-start justify-center p-4">
	<div class ="lg:w-2/4">
		<BlogList blogs={data.blogs} />
	</div>
	
</div>

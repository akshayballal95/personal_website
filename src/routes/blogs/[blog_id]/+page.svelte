<script lang="ts">
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string };

	let blog: any;
	$: blog = {};

	onMount(async () => {
		blog = await import(`../[blog_id]/blogs/${data.id}.md`);
	});

</script>

<svelte:head>
	<title>Blog</title>
	<meta name="description" content="Blogs written by Akshay Ballal" />
	<meta name="keywords" content="Artificial Intelligence, Machine Learning, Programming, Rust" />
</svelte:head>


<div class="flex justify-center p-10">
    <div class="flex flex-col gap-8 lg:w-2/4 bg-neutral-800 card p-5">
        <svelte:component this={blog.default} />
    
    </div>
</div>


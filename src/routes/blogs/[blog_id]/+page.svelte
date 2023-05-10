<script lang="ts">
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string, title: string };

	let blog: any;
	$: blog = {};


	onMount(async () => {
		blog = await import(`../[blog_id]/blogs/${data.id}.md`);
	});

</script>

<svelte:head>
	<title>{data.title}</title>
	<meta name="description" content="Blogs written by Akshay Ballal" />
	<meta name="keywords" content="Artificial Intelligence, Machine Learning, Programming, Rust" />
</svelte:head>

{#if blog}
<div class="flex justify-center lg:p-10 p-5">
    <div class="flex flex-col gap-8 w-full md:w-3/4 max-w-4xl rounded-md card md:p-8 p-5">
        <h2 class="mt-5">{data.title}</h2>

        <svelte:component this={blog.default} />
    
    </div>
</div>
{/if}


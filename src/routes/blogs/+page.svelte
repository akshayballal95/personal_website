<script lang="ts">
	import { goto } from '$app/navigation';
	import type { Blog } from '..//../input_model';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };
</script>

<svelte:head>
	<title>My Blogs</title>
	<meta name="description" content="Blogs written by Akshay Ballal" />
	<meta name="keywords" content="Artificial Intelligence, Machine Learning, Programming, Rust" />
</svelte:head>

<div class="flex flex-col p-6 gap-4 items-center justify-center">
	{#each data.blogs as blog}
		<div
			class="card lg:w-2/4 w-full flex fmd:flex-row p-4 gap-6 items-center justify-start md:h-40 h-auto"
		>
			<img class=" w-28 h-28 rounded-md object-fill" src={blog.image} alt="" />
			<div class="flex flex-col w-full gap-2">
				<p class="unstyled text-xs text-justify">{blog.date}</p>
				<a href = "/blogs/{blog.id}"><h2 class="unstyled font-bold">{blog.title}</h2></a>

				<p class="unstyled text-xs text-justify lg:line-clamp-4 line-clamp-5 overflow-hidden hidden md:block">
					{blog.description}
				</p>
				
				<!-- <button
					class="btn variant-filled-primary w-1/4 self-center lg:self-start mt-3 p-1"
					on:click={() => {
						goto(`/blogs/${blog.id}`);
					}}
				>
					Read
				</button> -->
			</div>
		</div>
	{/each}
</div>

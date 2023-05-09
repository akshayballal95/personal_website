<script lang="ts">
	import type { Blog } from '.././../input_model';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };
</script>

<div class="flex flex-col p-6 gap-4 items-center justify-center">
	{#each data.blogs as blog}
		<div class="card lg:w-3/4 w-full flex flex-col md:flex-row p-4 gap-6 items-center justify-start md:h-40 h-80  ">
			<img class=" w-28 h-28 rounded-md object-fill" src={blog.image} alt="" />
            <div class="flex flex-col w-full gap-2 ">
                <h2 class="unstyled text-sm font-bold">{blog.title}</h2>
                <p class="unstyled text-xs text-justify lg:line-clamp-4 line-clamp-5 overflow-hidden">{blog.description}</p>
                <button class="btn variant-filled-primary w-1/4 self-center lg:self-start mt-3 p-1"> Read </button>
            </div>
			
            
		</div>
	{/each}
</div>

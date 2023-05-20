<script lang="ts">
	import type { Blog } from '../../input_model';

	export let blogs: Blog[];
	function formattedDate (date:Date){
		return date.toLocaleDateString('en-GB', {
  day: 'numeric', month: 'short', year: 'numeric'
}).replace(/ /g, ' ');
	}
</script>

<div class="flex flex-col gap-4 overflow-y-scroll">
	{#each blogs as blog}
		<div
			class="card flex fmd:flex-row p-4 gap-6 items-center justify-start "
		>
			<img class=" w-28 h-28 rounded-md object-fill" src={blog.image} alt="" />
			<div class="flex flex-col w-full gap-2">
				<p class="unstyled text-xs text-justify">{formattedDate(blog.date)}</p>
				<a href="/blogs/{blog.id}"><h2 class="unstyled font-bold">{blog.title}</h2></a>

				<p
					class="unstyled text-xs text-justify lg:line-clamp-4 line-clamp-5 overflow-hidden hidden md:block"
				>
					{blog.description}
				</p>
			</div>
		</div>
	{/each}
</div>

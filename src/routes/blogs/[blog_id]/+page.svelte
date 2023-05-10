<script>
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';


	/** @type {import('./$types').PageData} */
	export let data;

	onMount(() => {
		$tabSet = 4;
	});
</script>

<!-- <div>
{data.data}
</div> -->
<div class="flex h-full justify-center items-center">
	<div class="flex flex-col gap-3 lg:w-3/4 p-10 bg-stone-800 ">{@html data.data}</div>
</div>
<!-- ![](https://res.cloudinary.com/dltwftrgc/image/upload/v1683653822/Blogs/AI_powered_game_bot/cover_image.png) -->
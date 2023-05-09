<script>
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';


    /** @type {import('./$types').PageData} */
    export let data

    onMount(() => {
		$tabSet = 4;
	});

</script>

<div class = "flex flex-col gap-3 h-full p-10">{@html data.data}</div>

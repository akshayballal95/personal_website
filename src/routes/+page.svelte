<script>
	import Aboutme from "$lib/aboutme/aboutme.svelte";
	import { tabSet } from "$lib/stores/stateStore";
	import { onMount } from "svelte";

	onMount(()=>{
		$tabSet = 0;
	}
	)

</script>
<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->

<Aboutme/>
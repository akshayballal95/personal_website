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

<svelte:head>
	<title> About Me</title>
	<meta name="I am passionate about building new technologies that push the boundaries of the status quo.
	My mix of knowledge in both mechanical systems and software development allows me to build
	more holistic systems and products. I have experience in building additive manufacturing
	machines and technologies and have put together several machines from scratch. I am the
	co-inventor of 7 granted patents in the field of composites and additive manufacturing.">

</svelte:head>

<Aboutme/>
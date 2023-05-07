<script>
	import Projects from "$lib/projects/projects.svelte";

	export let data
</script>

<svelte:head>
	<title>My Projects</title>
</svelte:head>

<Projects projects = {data}></Projects>
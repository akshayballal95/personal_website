<script lang = "ts">
    import { AppBar } from "@skeletonlabs/skeleton";
	import Navlinks from "./navlinks.svelte";
</script>

<AppBar>
	<svelte:fragment slot="lead">
		<div class="flex items-baseline">
			<h1 class="unstyled font-semibold text-2xl">Akshay Ballal</h1>
			<p class="unstyled font-light text-l">/CHIEF PRODUCT OFFICER</p>
		</div></svelte:fragment
	>

	<svelte:fragment slot="trail"><Navlinks></Navlinks></svelte:fragment>
    </AppBar
>

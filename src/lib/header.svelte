<script lang = "ts">
    import { AppBar } from "@skeletonlabs/skeleton";
	import Navlinks from "./navlinks.svelte";
</script>

<AppBar>
	<svelte:fragment slot="lead">
		<div class="flex items-baseline">
			<p class="unstyled font-bold text-2xl">Akshay Ballal</p>
			<p class="unstyled italic text-l">/CHIEF PRODUCT OFFICER</p>
		</div></svelte:fragment
	>

	<svelte:fragment slot="trail"><Navlinks></Navlinks></svelte:fragment>
    </AppBar
>

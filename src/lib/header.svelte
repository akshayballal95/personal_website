<script lang = "ts">
    import { AppBar } from "@skeletonlabs/skeleton";
	import Navlinks from "./navlinks.svelte";
</script>

<AppBar>
	<svelte:fragment slot="lead">
		<div class="md:flex-row flex flex-col items-baseline">
			<p class="unstyled font-bold text-2xl">Akshay Ballal</p>
			<span class="flex"><p class="unstyled italic text-l hidden lg:block">/</p><p class="unstyled italic text-l">CHIEF PRODUCT OFFICER</p></span>
		</div></svelte:fragment
	>

	<svelte:fragment slot="trail"><Navlinks></Navlinks></svelte:fragment>
    </AppBar
>

<script lang = "ts">
    import { AppBar } from "@skeletonlabs/skeleton";
	import Navlinks from "./navlinks.svelte";
	import { LightSwitch } from '@skeletonlabs/skeleton';
	import { goto } from "$app/navigation";
</script>


<AppBar>
	<svelte:fragment slot="lead" >
		<div class="md:flex-row flex flex-col items-baseline whitespace-nowrap cursor-pointer" on:click={()=>{goto("/")}} on:keydown={()=>{}}>
			<p class="unstyled font-bold text-2xl">Akshay Ballal</p>
			<span class="flex"><p class="unstyled italic text-l hidden md:block">/</p><p class="unstyled italic text-l">CHIEF PRODUCT OFFICER</p></span>
		</div></svelte:fragment
	>

	<svelte:fragment slot="trail"><Navlinks></Navlinks> <LightSwitch></LightSwitch></svelte:fragment>
    </AppBar
>

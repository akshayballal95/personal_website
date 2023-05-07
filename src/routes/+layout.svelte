<script lang='ts'>
	// The ordering of these imports is critical to your app working properly
	import '@skeletonlabs/skeleton/themes/theme-gold-nouveau.css';
	// If you have source.organizeImports set to true in VSCode, then it will auto change this ordering
	import '@skeletonlabs/skeleton/styles/all.css';
	// Most of your app wide CSS should be put in this file
	import '../app.postcss';
	
	import { AppShell } from '@skeletonlabs/skeleton';
	import Header from '$lib/header.svelte';
	import Footer from '$lib/footer.svelte';
	import {tabSet} from "$lib/stores/stateStore"
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';


	// onMount(()=>
	// {
	// if($tabSet == 0){
	// 	goto("/")
	// }
	// else if($tabSet == 1){
	// 	goto("/resume")
	// }


	// })
</script>


<AppShell>
	<svelte:fragment slot="header"
		><Header></Header></svelte:fragment
	>

	<!-- Router Slot -->
	<slot  />
	<!-- ---- / ---- -->
	<svelte:fragment slot="pageFooter"><Footer></Footer></svelte:fragment>
</AppShell>

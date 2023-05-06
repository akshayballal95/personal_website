<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->
<script>
	import { AppShell } from '@skeletonlabs/skeleton';
	import Header from '$lib/header.svelte';
	import { AppBar } from '@skeletonlabs/skeleton';
	import Footer from '$lib/footer.svelte';
</script>

<AppShell>
	<svelte:fragment slot="header"
		><Header></Header></svelte:fragment
	>

	<!-- Router Slot -->
	<slot />
	<!-- ---- / ---- -->
	<svelte:fragment slot="footer"><Footer></Footer></svelte:fragment>
</AppShell>

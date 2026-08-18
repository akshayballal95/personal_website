<script lang="ts">
	import { initializeStores } from '@skeletonlabs/skeleton';

	// Skeleton v2 moved drawer/toast/modal stores into context — they must be
	// initialized in the topmost layout before any child accesses them.
	initializeStores();
</script>

<slot />

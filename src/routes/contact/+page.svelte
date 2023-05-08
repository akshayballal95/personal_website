<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { toastStore, type ToastSettings } from '@skeletonlabs/skeleton';
	import { slide } from 'svelte/transition';

	/** @type {import('./$types').ActionData} */
	export let form: { success: string };

	const t: ToastSettings = {
		autohide: true,
		timeout: 1000,
		message: 'Message Sent Successfully',
		callback: (response) => console.log(response)
	};

	const e: ToastSettings = {
		autohide: true,
		timeout: 500,
		background: 'bg-error-500',

		message: 'Something went wrong. Please give all details',
		callback: (response) => console.log(response)
	};

	if (form?.success == 'true') {
		toastStore.trigger(t);
	} else if (form?.success == 'false') {
		toastStore.trigger(e);
	}

	interface FormData {
		firstName: string;
		lastName: string;
		email: string;
		subject: string;
		message: string;
	}
	let data: FormData;
	$: data = {
		firstName: '',
		lastName: '',
		email: '',
		subject: '',
		message: ''
	};

	$tabSet = 3;
</script>

<svelte:head>
	<title>Contact Me</title>
	<meta name="description" content="Contact Akshay" />
</svelte:head>

<div class="flex h-full justify-center">
	<form
		method="POST"
		class="card p-4 lg:min-w-[350px] lg:w-[500px] m-10 text-token space-y-4 flex flex-col gap-4"
		transition:slide
	>
		<h3 class = "self-center">Get in touch with me!</h3>
		<!-- Required -->
		<label class="label">
			<span>First Name</span>
			<input
				bind:value={data.firstName}
				class="input"
				name="firstName"
				type="text"
				placeholder="First Name"
			/>
		</label>
		<label class="label">
			<span>Last Name</span>
			<input
				bind:value={data.lastName}
				class="input"
				name="lastName"
				type="text"
				placeholder="Last Name"
			/>
		</label>
		<label class="label">
			<span>Email</span>
			<input bind:value={data.email} class="input" name="email" type="text" placeholder="Email" />
		</label>

		<label class="label">
			<span>Subject</span>
			<input
				bind:value={data.subject}
				class="input"
				name="subject"
				type="text"
				placeholder="Subject"
			/>
		</label>

		<label class="label">
			<span>Message</span>
			<!-- cspell:disable-next-line -->
			<textarea
				bind:value={data.message}
				class="textarea"
				name="message"
				rows="4"
				placeholder="Let me know how I can help you..."
			/>
		</label>

		<!-- If you want replyTo to be set to specific email -->
		<!-- <input type="text" name="replyTo" value="myreplytoemail@example.com" /> -->
		<!-- Optional -->
		<!-- If you want form to redirect to a specific url after submission -->
		<!-- <input type="hidden" name="redirectTo" value="https://example.com/contact/success"> Optional -->
		<!-- <input type="submit" value="Submit" /> -->

		<button type="submit" value="Submit" class="btn variant-filled"> SUBMIT </button>
	</form>
</div>

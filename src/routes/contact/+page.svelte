<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';

    interface FormData{
        firstName: String
        lastName: String
        email: String
        subject:String
        message:String
    }
    let data:FormData
    $: data = {
        firstName: "",
        lastName: "",
        email: "",
        subject: "",
        message: ""
    }
    

	$tabSet = 3;

    function submit(){
        console.log(data)
    }

</script>


<svelte:head>
    <title>Contact Me</title>
    <meta name="Contact Akshay">

</svelte:head>

<div class="flex h-full justify-center">
	<div class="card p-4 lg:min-w-[350px] lg:w-[500px] m-10 text-token space-y-4 flex flex-col gap-4">
		<label class="label">
			<span>First Name</span>
			<input bind:value={data.firstName} class="input" type="text" placeholder="First Name" />
		</label>
		<label class="label">
			<span>Last Name</span>
			<input bind:value={data.lastName} class="input" type="text" placeholder="Last Name" />
		</label>
		<label class="label">
			<span>Email</span>
			<input bind:value={data.email}  class="input" type="text" placeholder="Email" />
		</label>

		<label class="label">
			<span>Subject</span>
			<input bind:value={data.subject}  class="input" type="text" placeholder="Subject" />
		</label>

		<label class="label">
			<span>Message</span>
			<!-- cspell:disable-next-line -->
			<textarea bind:value={data.message}  class="textarea" rows="4" placeholder="Let me know how I can help you..." />
		</label>

		<button on:click={submit} class="btn variant-filled"> SUBMIT </button>
	</div>
</div>

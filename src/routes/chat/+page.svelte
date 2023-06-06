<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { Avatar } from '@skeletonlabs/skeleton';

	class chatInterface {
		constructor() {
			this.history = '';
			this.human_input = '';
			this.memory = [];
		}

		history: String;
		human_input: String;
		memory: memory[];
	}

	class memory {
		constructor() {
			this.type = '';
			this.data = { content: '', additional_kwargs: {} };
		}

		type: String;
		data: {
			content: String;
			additional_kwargs: {};
		};
	}
    let chatMessage = new chatInterface()
	$: chatMessage: chatInterface;

	async function chat(message: String) {
		let body: chatInterface = {
			history: 'string',
			human_input: message,
			memory: []
		};

		const response = await fetch('http://13.53.243.0/app.conversation/run', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				history: 'string',
				human_input: message,
				memory: [{}]
			})
		});
		console.log(response);
		return response;
	}

	$tabSet = 5;
</script>

<button class="btn variant-filled" on:click={() => chat('How are you')}>Chat</button>
<div class="flex w-full justify-center">
    <div class="grid grid-cols-[auto_1fr] gap-2 w-2/4">
        <!-- <Avatar src="https://i.pravatar.cc/?img={}" width="w-12" /> -->
        <div class="card p-4 variant-soft rounded-tl-none space-y-2">
            <header class="flex justify-between items-center">
                <p class="font-bold">{"Akshay"}</p>
                <!-- <small class="opacity-50">{bubble.timestamp}</small> -->
            </header>
            <p>{"Debitis dolores adipisci. Quia hic quae nostrum quibusdam veniam. Necessitatibus debitis reprehenderit. Iste illo voluptas cumque exercitationem repellendus eius eum laborum earum. Perspiciatis reprehenderit accusamus magni nihil fugiat tempore."}</p>
        </div>
    </div>
</div>


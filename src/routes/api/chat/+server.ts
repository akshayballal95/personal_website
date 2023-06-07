import { error } from '@sveltejs/kit';

/** @type {import('./$types').RequestHandler} */
export async function POST({ request }) {
    let body = await request.json()
    const response = await fetch('https://gpt_server-1-v2639104.deta.app/app.conversation/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept':'application/json',
            'X-Space-App-Key':'QcmgCuZ7qSfavrMBZHoxrMC3VyDrTKTN'
        },
        body: JSON.stringify(body),
    });

    let result = await response.json()


    return new Response(JSON.stringify(result))
}
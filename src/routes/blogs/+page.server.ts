
import { Client } from "@notionhq/client"
import type {PageObjectResponse} from "@notionhq/client/build/src/api-endpoints"
import { VITE_NOTION_KEY } from "$env/static/private"
import { VITE_DATABASE_ID } from "$env/static/private"
import Page from "../+page.svelte"
import type {Blog}  from ".././../input_model"




/** @type {import('./$types').PageLoad} */
export async function load({params}){

    const notion = new Client({ auth: VITE_NOTION_KEY })
    const databaseId = VITE_DATABASE_ID
    const response = await notion.databases.query({
        filter:{
            property:"Status",
            select:{
                equals:"Live"
            }
        },
        database_id: databaseId,
    })


    let blogs:Blog[] = []
    response.results.forEach( async (element) => {
        let elem = element as any
        let blog:Blog = {
            title: elem.properties.Name.title[0].text.content,
            image : elem.cover? elem.cover.file.url:null,
            description : elem.properties.Description? elem.properties.Description.rich_text[0].text.content:null,
        }
  
        blogs.push(blog)


    });

    return ({
        blogs
    })

}
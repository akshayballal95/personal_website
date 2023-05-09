
import { Client } from "@notionhq/client"
import { VITE_NOTION_KEY } from "$env/static/private"
import { VITE_DATABASE_ID } from "$env/static/private"
import type {Blog}  from "../../input_model"




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

    try{
        response.results.forEach( async (element) => {
        
            let elem = element as any
            console.log(elem.properties.Description)
            let blog:Blog = {
                title: elem.properties.Name.title[0].text.content,
                image : elem.cover!=null? elem.cover.file.url:null,
                description : elem.properties.Description.rich_text[0]? elem.properties.Description.rich_text[0].text.content:"No description provided",
                date: elem.properties.Date.date?elem.properties.Date.date.start:"",
                id: element.id
            }
      
            blogs.push(blog)
    
    
        });
    
        return ({
            blogs
        })
    }
    catch(e){console.log(e)}


}
// import { graphql } from "gatsby"
import React from "react"
import Layout from "../components/Layout"

export default function Home() {
	// console.log(data)
	// const { title, description } = data.site.siteMetadata
	return (
		<Layout>
			<section>
				<div>Hello world!</div>
			</section>
		</Layout>
	)
}
// export const query = graphql`
// 	query SiteInfo {
// 		site {
// 			id
// 			siteMetadata {
// 				description
// 				title
// 			}
// 		}
// 	}
// `

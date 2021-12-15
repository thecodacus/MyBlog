import React from "react"
import Layout from "../components/Layout"
import PageHeroName from "../components/PageHeroName"

export default function NotFound() {
	return (
		<Layout>
			<PageHeroName>404 not found</PageHeroName>
			{/* <p>Sorry this page does not exist</p> */}
		</Layout>
	)
}

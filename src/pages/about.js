import React from "react"
import Layout from "../components/Layout"
import PageHeroName from "../components/PageHeroName"
import Seo from "../components/Seo"
export default function About() {
	return (
		<Layout>
			<Seo title="About" />
			<PageHeroName>About</PageHeroName>
		</Layout>
	)
}

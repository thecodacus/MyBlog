import React from "react"
import Layout from "../components/Layout"
import Img from "gatsby-image"
import * as styles from "../styles/post-details.module.css"
import { graphql } from "gatsby"

export default function ProjectDetails({ data }) {
	console.log(data)
	const { html } = data.markdownRemark
	const { title, category, featuredImage } = data.markdownRemark.frontmatter

	return (
		<Layout>
			<div className={styles.details}>
				<h2>{title}</h2>
				<h3>{category}</h3>
				<div className={styles.featured}>
					<Img fluid={featuredImage.childImageSharp.fluid} />
				</div>
				<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>
			</div>
		</Layout>
	)
}

export const query = graphql`
	query PostDetails($slug: String) {
		markdownRemark(frontmatter: { slug: { eq: $slug } }) {
			id
			html
			frontmatter {
				date
				slug
				category
				title
				featuredImage {
					childImageSharp {
						fluid {
							...GatsbyImageSharpFluid
						}
					}
				}
			}
		}
	}
`

import React from "react"
import Layout from "../components/Layout"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import * as styles from "../styles/post-details.module.scss"
import { graphql } from "gatsby"

export default function PostDetails({ data }) {
	console.log(data)
	const { html } = data.markdownRemark
	const { title, category, featuredImage, date } = data.markdownRemark.frontmatter

	return (
		<Layout>
			<div className={styles.details}>
				<h1>{title}</h1>
				<div className={styles.featured}>
					<GatsbyImage image={getImage(featuredImage)} alt={title} />
				</div>
				<div className={styles.metadata}>
					<h3>{new Date(date).toDateString()}</h3>
					<h3>.</h3>
					<h3>{category}</h3>
				</div>
				<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>
			</div>
		</Layout>
	)
}

export const query = graphql`
	query PostDetails($slug: String) {
		markdownRemark(fields: { slug: { eq: $slug } }) {
			id
			html
			fields {
				slug
			}
			frontmatter {
				date
				slug
				category
				title
				featuredImage {
					childImageSharp {
						gatsbyImageData(layout: FULL_WIDTH)
					}
				}
			}
		}
	}
`

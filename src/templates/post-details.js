import React from "react"
import { graphql } from "gatsby"
import Disqus from "gatsby-plugin-disqus"
import Layout from "../components/Layout"
import { GatsbyImage, getImage, getSrc } from "gatsby-plugin-image"
import * as styles from "../styles/post-details.module.scss"

export default function PostDetails({ data }) {
	console.log(data)
	const {
		html,
		fields: { slug },
	} = data.markdownRemark
	const { title, category, featuredImage, date } = data.markdownRemark.frontmatter
	const featuredImagePath = featuredImage.publicURL

	return (
		<Layout>
			<div className={styles.details}>
				<h1>{title}</h1>
				<div className={styles.featured}>
					{featuredImage.childImageSharp != null ? (
						<GatsbyImage image={getImage(featuredImage)} alt={title} />
					) : (
						<img className={styles.alternateImage} src={featuredImagePath} alt={title} />
					)}
				</div>
				<div className={styles.metadata}>
					<h3>{new Date(date).toDateString()}</h3>
					<h3>.</h3>
					<h3>{category}</h3>
				</div>
				<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>
				<Disqus identifier={slug} title={title} />
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
					publicURL
					childImageSharp {
						gatsbyImageData(layout: FULL_WIDTH)
					}
				}
			}
		}
	}
`

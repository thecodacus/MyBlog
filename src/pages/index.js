// import { graphql } from "gatsby"
import { graphql } from "gatsby"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import React from "react"
import Layout from "../components/Layout"
import PostsGrid from "../components/PostsGrid"
import * as styles from "../styles/home.module.scss"

export default function Home({ data }) {
	console.log(data)
	const { banner, info, posts } = data
	return (
		<Layout>
			<div className={styles.home}>
				<section className={styles.hero}>
					<div className={styles.heroText}>
						<div className={styles.title}>Welcome To</div>
						<div className={styles.title}>{info.siteMetadata.subHeader}</div>
						<div className={styles.description}>{info.siteMetadata.description}</div>
					</div>
					<GatsbyImage image={getImage(banner)} alt="banner image" />
				</section>
				<section className="postgrid">
					<PostsGrid posts={posts.nodes} />
				</section>
			</div>
		</Layout>
	)
}
export const query = graphql`
	query HomePageContent {
		banner: file(relativePath: { eq: "homepage-banner.jpeg" }) {
			childImageSharp {
				gatsbyImageData(layout: FULL_WIDTH)
			}
		}
		info: site {
			siteMetadata {
				title
				subHeader
				description
			}
		}
		posts: allMarkdownRemark(sort: { fields: frontmatter___date, order: DESC }) {
			nodes {
				fields {
					slug
				}
				excerpt
				frontmatter {
					title
					category
					date
					featuredImage {
						publicURL
						childImageSharp {
							gatsbyImageData(aspectRatio: 1.5, layout: FULL_WIDTH)
						}
					}
				}
				id
			}
		}
	}
`

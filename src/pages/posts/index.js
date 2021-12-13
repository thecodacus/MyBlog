import { graphql, Link } from "gatsby"
import React from "react"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import Layout from "../../components/Layout"
import * as styles from "../../styles/posts.module.scss"
export default function Posts({ data }) {
	console.log(data)
	const posts = data.posts.nodes
	return (
		<Layout>
			<section>
				<div className={styles.posts}>
					{posts.map(post => (
						<Link to={"/posts/" + post.fields.slug} key={post.id}>
							<div>
								<div className="thumbnail">
									<GatsbyImage image={getImage(post.frontmatter.featuredImage)} alt={post.frontmatter.title} />
								</div>
								<h3>{post.frontmatter.title}</h3>
								<p>{post.frontmatter.category}</p>
							</div>
						</Link>
					))}
				</div>
			</section>
		</Layout>
	)
}

export const query = graphql`
	query AllPosts {
		posts: allMarkdownRemark(sort: { fields: frontmatter___date, order: DESC }) {
			nodes {
				fields {
					slug
				}
				frontmatter {
					title
					slug
					category
					date
					featuredImage {
						childImageSharp {
							gatsbyImageData(layout: FULL_WIDTH)
						}
					}
				}
				id
			}
		}
	}
`
